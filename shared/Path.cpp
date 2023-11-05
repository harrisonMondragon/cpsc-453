#include "Path.hpp"

#include <cassert>
#include <filesystem>

#define VALUE(string) #string
#define TO_LITERAL(string) VALUE(string)

namespace shared
{

	//-------------------------------------------------------------------------------------------------

	std::shared_ptr<Path> Path::Instantiate()
	{
		return std::make_shared<Path>();
	}

	//-------------------------------------------------------------------------------------------------

	Path::Path()
	{
		Instance = this;

#if defined(ASSET_DIR)
		mAssetPath = std::filesystem::absolute(std::string(TO_LITERAL(ASSET_DIR))).string();
#else
#error Asset directory is not defined
#endif
	}

	//-------------------------------------------------------------------------------------------------

	Path::~Path()
	{
		assert(Instance != nullptr);
		Instance = nullptr;
	}

	//-------------------------------------------------------------------------------------------------

	std::string Path::Get(std::string const& address) const
	{
		return std::filesystem::path(mAssetPath).append(address).string();
	}

	//-------------------------------------------------------------------------------------------------

}