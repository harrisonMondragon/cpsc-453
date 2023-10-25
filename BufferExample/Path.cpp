#include "Path.hpp"

#include <cassert>
#include <filesystem>

#define VALUE(string) #string
#define TO_LITERAL(string) VALUE(string)

//-------------------------------------------------------------------------------------------------

std::shared_ptr<BufferExample::Path> BufferExample::Path::Instantiate()
{
	return std::make_shared<Path>();
}

//-------------------------------------------------------------------------------------------------

BufferExample::Path::Path()
{
	Instance = this;

#if defined(ASSET_DIR)
	mAssetPath = std::filesystem::absolute(std::string(TO_LITERAL(ASSET_DIR))).string();
#else
#error Asset directory is not defined
#endif
}

//-------------------------------------------------------------------------------------------------

BufferExample::Path::~Path()
{
	assert(Instance != nullptr);
	Instance = nullptr;
}

//-------------------------------------------------------------------------------------------------

std::string BufferExample::Path::Get(std::string const& address) const
{
	return std::filesystem::path(mAssetPath).append(address).string();
}

//-------------------------------------------------------------------------------------------------
